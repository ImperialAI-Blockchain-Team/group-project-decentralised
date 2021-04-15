import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0x2F2dbC0ca7Bf1390196DCE21B595BDA29834B6C5';

export default new web3.eth.Contract(jobs, address);